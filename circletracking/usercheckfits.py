from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import pandas
import matplotlib.pyplot as plt


class UserCheckFits(object):
    """
    Let user manually check fitted circles using an interface in Matplotlib
    """
    # Store fits so that we can use it in event callback
    stored_fits = pandas.DataFrame()

    def __init__(self, filename, micron_per_pixel):
        """
        Let user manually check fits, removing them by clicking
        :return:
        """
        UserCheckFits.stored_fits = pandas.DataFrame.from_csv(filename)

        # Set scale in pixels
        UserCheckFits.stored_fits /= micron_per_pixel

        UserCheckFits.stored_fits['remove'] = False

    def user_check_fits(self, image):
        """

        :param image:
        :return:
        """
        self.plot_fits_for_user_confirmation(image)

        mask = (UserCheckFits.stored_fits['remove'] == False)
        UserCheckFits.stored_fits = UserCheckFits.stored_fits[mask]

        UserCheckFits.stored_fits.drop('remove', axis=1, inplace=True)

        # Update indices
        UserCheckFits.stored_fits.reset_index(inplace=True)

        return UserCheckFits.stored_fits

    @classmethod
    def on_pick(cls, event):
        """
        User clicked on a fit
        :param event:
        """
        # Get index from label
        fit_number = int(event.artist.get_label())

        if not UserCheckFits.stored_fits['remove'][fit_number]:
            event.artist.set_edgecolor('r')
            cls.set_annotation_color(fit_number, 'r')
            UserCheckFits.stored_fits.set_value(fit_number, 'remove', True)
        else:
            event.artist.set_edgecolor('b')
            cls.set_annotation_color(fit_number, 'b')
            UserCheckFits.stored_fits.set_value(fit_number, 'remove', False)

        event.canvas.draw()

    def plot_fits_for_user_confirmation(self, image):
        """
        Ask user to check if all fits are correct.
        Clicking on a fit removes it from the results.
        :param image:
        """
        _imshow_style = dict(origin='lower', interpolation='none',
                             cmap=plt.cm.gray)
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.showMaximized()
        plt.clf()
        plt.cla()
        plt.imshow(image, **_imshow_style)
        for i in UserCheckFits.stored_fits.index:
            circle = plt.Circle((UserCheckFits.stored_fits.loc[i].x,
                                 UserCheckFits.stored_fits.loc[i].y),
                                radius=UserCheckFits.stored_fits.loc[i].r,
                                fc='None', ec='b', ls='solid',
                                lw=0.3, label=i)

            # Enable picking
            circle.set_picker(10)

            plt.gca().add_patch(circle)

            plt.gca().annotate(i, (UserCheckFits.stored_fits.loc[i].x,
                                   UserCheckFits.stored_fits.loc[i].y),
                               color='b', weight='normal', size=8,
                               ha='center', va='center')
            plt.gca().set_title('Please check the result.'
                                + ' Click on a circle to toggle removal.'
                                + ' Close to confirm.')
            plt.gcf().canvas.mpl_connect('pick_event', self.on_pick)
        plt.show(block=True)

    @staticmethod
    def set_annotation_color(index, color):
        """
        Remove particle index from the current plot
        :param index:
        """
        children = plt.gca().get_children()
        children = [c for c in children if isinstance(c, plt.Annotation)
                    and int(c._text) == index]

        for child in children:
            child.set_color(color)
